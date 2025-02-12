import typing

from pydantic import field_validator
from pydantic import PostgresDsn
from pydantic import ValidationInfo
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "maze"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DB: str = "maze"
    DATABASE_URL: typing.Optional[PostgresDsn] = None

    @field_validator("DATABASE_URL", mode="before")
    def assemble_db_connection(
        cls, v: typing.Optional[str], info: ValidationInfo
    ) -> typing.Any:
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql",
            username=info.data.get("POSTGRES_USER"),
            password=info.data.get("POSTGRES_PASSWORD"),
            host=info.data.get("POSTGRES_SERVER"),
            path=f"{info.data.get('POSTGRES_DB') or ''}",
        )


# Do not import and access this directly, use settings instead
_settings = Settings()


class SettingsProxy:
    def __init__(self, get_settings: typing.Callable[[], Settings]):
        self._get_settings = get_settings

    def __getattr__(self, item: str) -> typing.Any:
        global_settings = self._get_settings()
        return getattr(global_settings, item)


settings: Settings = SettingsProxy(lambda: _settings)
